import base64
import os
from urllib import error
from urllib import parse
from urllib import request
from openstack import exceptions
def resource_nested_identifier(rsrc):
    nested_link = [link for link in rsrc.links or [] if link.get('rel') == 'nested']
    if nested_link:
        nested_href = nested_link[0].get('href')
        nested_identifier = nested_href.split('/')[-2:]
        return '/'.join(nested_identifier)