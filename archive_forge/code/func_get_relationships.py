import json
import re
import zipfile
from abc import ABC
from pathlib import Path
from typing import Iterator, List, Set, Tuple
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def get_relationships(self, page: str, zfile: zipfile.ZipFile, filelist: List[str], pagexml_rels: List[dict]) -> Set[str]:
    """Get the relationships of a page and the relationships of its relationships,
        etc... recursively.
        Pages are based on other pages (ex: background page),
        so we need to get all the relationships to get all the content of a single page.
        """
    name_path = Path(page).name
    parent_path = Path(page).parent
    rels_path = parent_path / f'_rels/{name_path}.rels'
    if str(rels_path) not in zfile.namelist():
        return set()
    pagexml_rels_content = next((page_['content'] for page_ in pagexml_rels if page_['path'] == page))
    if isinstance(pagexml_rels_content['Relationships']['Relationship'], list):
        targets = [rel['@Target'] for rel in pagexml_rels_content['Relationships']['Relationship']]
    else:
        targets = [pagexml_rels_content['Relationships']['Relationship']['@Target']]
    relationships = set([str(parent_path / target) for target in targets]).intersection(filelist)
    for rel in relationships:
        relationships = relationships | self.get_relationships(rel, zfile, filelist, pagexml_rels)
    return relationships