import collections
import email.message
import functools
import itertools
import json
import logging
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from optparse import Values
from typing import (
from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.vcs import vcs
from .sources import CandidatesFromPage, LinkSource, build_source
def collect_sources(self, project_name: str, candidates_from_page: CandidatesFromPage) -> CollectedSources:
    index_url_sources = collections.OrderedDict((build_source(loc, candidates_from_page=candidates_from_page, page_validator=self.session.is_secure_origin, expand_dir=False, cache_link_parsing=False, project_name=project_name) for loc in self.search_scope.get_index_urls_locations(project_name))).values()
    find_links_sources = collections.OrderedDict((build_source(loc, candidates_from_page=candidates_from_page, page_validator=self.session.is_secure_origin, expand_dir=True, cache_link_parsing=True, project_name=project_name) for loc in self.find_links)).values()
    if logger.isEnabledFor(logging.DEBUG):
        lines = [f'* {s.link}' for s in itertools.chain(find_links_sources, index_url_sources) if s is not None and s.link is not None]
        lines = [f'{len(lines)} location(s) to search for versions of {project_name}:'] + lines
        logger.debug('\n'.join(lines))
    return CollectedSources(find_links=list(find_links_sources), index_urls=list(index_url_sources))