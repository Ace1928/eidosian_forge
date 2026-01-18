import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def post_form(self, orig_response, **kwargs):
    """
        The same as select_form but with no possibility of change the content
        of the form.

        :param httpc: A HTTP Client instance
        :param orig_response: The original response (as returned by requests)
        :param content: The content of the response
        :return: The response do_click() returns
        """
    response = RResponse(orig_response)
    form = self.pick_form(response, **kwargs)
    return self.do_click(form, **kwargs)