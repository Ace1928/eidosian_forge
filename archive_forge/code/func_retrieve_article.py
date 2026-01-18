import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def retrieve_article(self, uid: str, webenv: str) -> dict:
    url = self.base_url_efetch + 'db=pubmed&retmode=xml&id=' + uid + '&webenv=' + webenv
    retry = 0
    while True:
        try:
            result = urllib.request.urlopen(url)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and retry < self.max_retry:
                print(f'Too Many Requests, waiting for {self.sleep_time:.2f} seconds...')
                time.sleep(self.sleep_time)
                self.sleep_time *= 2
                retry += 1
            else:
                raise e
    xml_text = result.read().decode('utf-8')
    text_dict = self.parse(xml_text)
    return self._parse_article(uid, text_dict)