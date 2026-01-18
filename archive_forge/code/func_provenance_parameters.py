import time
import platform
import socket
from lxml import etree
from lxml.etree import Element, QName
from .uriutil import uri_parent
from .jsonutil import JsonTable
from . import httputil
def provenance_parameters(process_steps):
    prov = []
    for step in process_steps:
        if not set(_required).issubset(step.keys()):
            missing = list(set(_required).difference(step.keys()))
            raise Exception('Following attributes are required to define provenance: %s' % missing)
        prov.append(process_step_xml(**step))
    return prov