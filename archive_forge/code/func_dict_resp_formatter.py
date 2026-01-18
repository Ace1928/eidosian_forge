from collections import namedtuple
import json
import logging
import pprint
import re
@classmethod
def dict_resp_formatter(cls, resp):
    return resp.value