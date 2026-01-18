import os
import tempfile
import urllib.parse
from typing import Any, List, Optional
from urllib.parse import urljoin
import requests
from langchain_core.documents import Document
from requests.auth import HTTPBasicAuth
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
def ls_objects(self, repo: str, ref: str, path: str, presign: Optional[bool]) -> List:
    qp = {'prefix': path, 'presign': presign}
    eqp = urllib.parse.urlencode(qp)
    objects_ls_endpoint = urljoin(self.__endpoint, f'repositories/{repo}/refs/{ref}/objects/ls?{eqp}')
    olsr = requests.get(objects_ls_endpoint, auth=self.__auth)
    olsr.raise_for_status()
    olsr_json = olsr.json()
    return list(map(lambda res: (res['path'], res['physical_address']), olsr_json['results']))