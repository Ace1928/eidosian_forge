from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def authenticate_session(self):
    """
        Performs session authentication with Avi controller and stores
        session cookies and sets header options like tenant.
        """
    body = {'username': self.avi_credentials.username}
    if self.avi_credentials.password:
        body['password'] = self.avi_credentials.password
    elif self.avi_credentials.token:
        body['token'] = self.avi_credentials.token
    else:
        raise APIError('Neither user password or token provided')
    logger.debug('authenticating user %s prefix %s', self.avi_credentials.username, self.prefix)
    self.cookies.clear()
    err = None
    try:
        rsp = super(ApiSession, self).post(self.prefix + '/login', body, timeout=self.timeout, verify=self.verify)
        if rsp.status_code == 200:
            self.num_session_retries = 0
            self.remote_api_version = rsp.json().get('version', {})
            self.session_cookie_name = rsp.json().get('session_cookie_name', 'sessionid')
            self.headers.update(self.user_hdrs)
            if rsp.cookies and 'csrftoken' in rsp.cookies:
                csrftoken = rsp.cookies['csrftoken']
                sessionDict[self.key] = {'csrftoken': csrftoken, 'session_id': rsp.cookies[self.session_cookie_name], 'last_used': datetime.utcnow(), 'api': self, 'connected': True}
            logger.debug('authentication success for user %s', self.avi_credentials.username)
            return
        elif rsp.status_code in [401, 403]:
            logger.error('Status Code %s msg %s', rsp.status_code, rsp.text)
            err = APIError('Status Code %s msg %s' % (rsp.status_code, rsp.text), rsp)
            raise err
        else:
            logger.error('Error status code %s msg %s', rsp.status_code, rsp.text)
            err = APIError('Status Code %s msg %s' % (rsp.status_code, rsp.text), rsp)
    except (RequestsConnectionError, SSLError) as e:
        if not self.retry_conxn_errors:
            raise
        logger.warning('Connection error retrying %s', e)
        err = e
    if self.retry_wait_time:
        time.sleep(self.retry_wait_time)
    self.num_session_retries += 1
    if self.num_session_retries > self.max_session_retries:
        self.num_session_retries = 0
        logger.error('giving up after %d retries connection failure %s', self.max_session_retries, True)
        ret_err = err if err else APIError('giving up after %d retries connection failure %s' % (self.max_session_retries, True))
        raise ret_err
    self.authenticate_session()
    return