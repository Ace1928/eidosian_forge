import json
from xml.dom import pulldom
from xml.sax import handler
from xml.sax.expatreader import ExpatParser as _ExpatParser
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.xmlutils import SimplerXMLGenerator, UnserializableContentError
def entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name):
    raise EntitiesForbidden(name, value, base, sysid, pubid, notation_name)