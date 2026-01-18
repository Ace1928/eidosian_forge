from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from http import client as http_client_lib
import json
import logging
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional, Union
import urllib
from absl import flags
import googleapiclient
from googleapiclient import discovery
import httplib2
import bq_flags
import bq_utils
from clients import bigquery_http
from clients import utils as bq_client_utils
from discovery_documents import discovery_document_cache
from discovery_documents import discovery_document_loader
from utils import bq_api_utils
from utils import bq_error
@property
def apiclient(self) -> discovery.Resource:
    """Returns a singleton ApiClient built for the BigQuery core API."""
    if self._apiclient:
        logging.info('Using the cached BigQuery API client')
    else:
        self._apiclient = self.BuildApiClient(service=Service.BIGQUERY)
    return self._apiclient