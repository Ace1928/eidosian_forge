from __future__ import absolute_import, division, print_function
import json
from collections import defaultdict
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.parse import quote as urlquote
def query_deployment(self, configs=None, activities=None):
    """
        Find the current deployment or the deployment specified with configs or activities
        in the director and return the result of the api-call.

        Parameters:
            configs: type list, default empty, list of checksums for configs to search for.
                     If left empty, only the active_configuration will be returned.
            activities: type list, default empty, list of checksums for activities to search for.
                     If left empty, only the active_configuration will be returned.

        Returns:
            the result of the api-call
        """
    if configs is None:
        configs = []
    if activities is None:
        activities = []
    try:
        ret = self.call_url(path=self.path + '?configs=' + ','.join(configs) + '&activities=' + ','.join(activities))
        if ret['code'] != 200:
            self.module.fail_json(msg='bad return code while querying: %d. Error message: %s' % (ret['code'], ret['error']))
        return ret
    except Exception as e:
        self.module.fail_json(msg='exception when querying: ' + str(e))