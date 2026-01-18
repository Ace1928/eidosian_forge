import os
import re
from typing import Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def getNormTx(self) -> List[Document]:
    url = f'https://api.etherscan.io/api?module=account&action=txlist&address={self.account_address}&startblock={self.start_block}&endblock={self.end_block}&page={self.page}&offset={self.offset}&sort={self.sort}&apikey={self.api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print('Error occurred while making the request:', e)
    items = response.json()['result']
    result = []
    if len(items) == 0:
        return [Document(page_content='')]
    for item in items:
        content = str(item)
        metadata = {'from': item['from'], 'tx_hash': item['hash'], 'to': item['to']}
        result.append(Document(page_content=content, metadata=metadata))
    print(len(result))
    return result