import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def select_form(self, orig_response, **kwargs):
    """
        Pick a form on a web page, possibly enter some information and submit
        the form.

        :param orig_response: The original response (as returned by requests)
        :return: The response do_click() returns
        """
    logger.info('select_form')
    response = RResponse(orig_response)
    try:
        _url = response.url
    except KeyError:
        _url = kwargs['location']
    form = self.pick_form(response, _url, **kwargs)
    if not form:
        raise Exception("Can't pick a form !!")
    if 'set' in kwargs:
        for key, val in kwargs['set'].items():
            if key.startswith('_'):
                continue
            if 'click' in kwargs and kwargs['click'] == key:
                continue
            try:
                form[key] = val
            except ControlNotFoundError:
                pass
            except TypeError:
                cntrl = form.find_control(key)
                if isinstance(cntrl, ListControl):
                    form[key] = [val]
                else:
                    raise
    if form.action in kwargs['conv'].my_endpoints():
        return {'SAMLResponse': form['SAMLResponse'], 'RelayState': form['RelayState']}
    return self.do_click(form, **kwargs)