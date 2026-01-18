import lxml
import os
import os.path as op
import sys
import re
import shutil
import tempfile
import zipfile
import codecs
from fnmatch import fnmatch
from itertools import islice
from lxml import etree
from pathlib import Path
from .uriutil import join_uri, translate_uri, uri_segment
from .uriutil import uri_last, uri_nextlast
from .uriutil import uri_parent, uri_grandparent
from .uriutil import uri_shape
from .uriutil import file_path
from .jsonutil import JsonTable, get_selection
from .pathutil import find_files, ensure_dir_exists
from .attributes import EAttrs
from .search import rpn_contraints, query_from_xml
from .errors import is_xnat_error, parse_put_error_message
from .errors import DataError, ProgrammingError, catch_error
from .provenance import Provenance
from .pipelines import Pipelines
from . import schema
from . import httputil
from . import downloadutils
from . import derivatives
import types
import pkgutil
import inspect
from urllib.parse import quote, unquote
def get_custom_variables(self):
    """Retrieves custom variables as a dictionary

        It has the format {studyProtocol: { setname : {field: type, ...}}}

        """
    tree = lxml.etree.fromstring(self.get())
    nsmap = tree.nsmap
    custom_variables = {}
    for studyprotocols in tree.xpath('//xnat:studyProtocol', namespaces=nsmap):
        protocol_name = studyprotocols.get('name')
        custom_variables[protocol_name] = {}
        for definition in studyprotocols.xpath('xnat:definitions/xnat:definition', namespaces=nsmap):
            definition_id = definition.get('ID')
            custom_variables[protocol_name][definition_id] = {}
            for field in definition.xpath('xnat:fields/xnat:field', namespaces=nsmap):
                field_name = field.get('name')
                if field.get('type') == 'custom':
                    custom_variables[protocol_name][definition_id][field_name] = field.get('datatype')
    return custom_variables