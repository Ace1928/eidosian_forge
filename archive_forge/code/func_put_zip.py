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
def put_zip(self, zip_location, overwrite=False, extract=True, **datatypes):
    """ Uploads a zip or tgz file an then extracts it on the server.

            After the compressed file is extracted the individual
            files are accessible separately, or as a whole using get_zip.

            Parameters
            ----------
            zip_location: Path to zip file for upload.

            overwrite: boolean
                If True, overwrite the files that already exist under the
                    given id.
                If False, do not overwrite (Default)

            extract: boolean
                If True, the uploaded zip file is extracted. (Default)
                If False, the file is not extracted.
        """
    if not self.exists():
        self.create(**datatypes)
    if extract:
        do_extract = '?extract=true'
    else:
        do_extract = ''
    self.file(op.split(zip_location)[1] + do_extract).put(zip_location, overwrite=overwrite, **datatypes)