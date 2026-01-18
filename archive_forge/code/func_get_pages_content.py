import json
import re
import zipfile
from abc import ABC
from pathlib import Path
from typing import Iterator, List, Set, Tuple
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def get_pages_content(self, zfile: zipfile.ZipFile, source: str) -> List[Tuple[int, str, str]]:
    """Get the content of the pages of a vsdx file.

        Attributes:
            zfile (zipfile.ZipFile): The vsdx file under zip format.
            source (str): The path of the vsdx file.

        Returns:
            list[tuple[int, str, str]]: A list of tuples containing the page number,
            the name of the page and the content of the page
            for each page of the vsdx file.
        """
    try:
        import xmltodict
    except ImportError:
        raise ImportError('The xmltodict library is required to parse vsdx files. Please install it with `pip install xmltodict`.')
    if 'visio/pages/pages.xml' not in zfile.namelist():
        print('WARNING - No pages.xml file found in {}'.format(source))
        return
    if 'visio/pages/_rels/pages.xml.rels' not in zfile.namelist():
        print('WARNING - No pages.xml.rels file found in {}'.format(source))
        return
    if 'docProps/app.xml' not in zfile.namelist():
        print('WARNING - No app.xml file found in {}'.format(source))
        return
    pagesxml_content: dict = xmltodict.parse(zfile.read('visio/pages/pages.xml'))
    appxml_content: dict = xmltodict.parse(zfile.read('docProps/app.xml'))
    pagesxmlrels_content: dict = xmltodict.parse(zfile.read('visio/pages/_rels/pages.xml.rels'))
    if isinstance(pagesxml_content['Pages']['Page'], list):
        disordered_names: List[str] = [rel['@Name'].strip() for rel in pagesxml_content['Pages']['Page']]
    else:
        disordered_names: List[str] = [pagesxml_content['Pages']['Page']['@Name'].strip()]
    if isinstance(pagesxmlrels_content['Relationships']['Relationship'], list):
        disordered_paths: List[str] = ['visio/pages/' + rel['@Target'] for rel in pagesxmlrels_content['Relationships']['Relationship']]
    else:
        disordered_paths: List[str] = ['visio/pages/' + pagesxmlrels_content['Relationships']['Relationship']['@Target']]
    ordered_names: List[str] = appxml_content['Properties']['TitlesOfParts']['vt:vector']['vt:lpstr'][:len(disordered_names)]
    ordered_names = [name.strip() for name in ordered_names]
    ordered_paths = [disordered_paths[disordered_names.index(name.strip())] for name in ordered_names]
    disordered_pages = []
    for path in ordered_paths:
        content = zfile.read(path)
        string_content = json.dumps(xmltodict.parse(content))
        samples = re.findall('"#text"\\s*:\\s*"([^\\\\"]*(?:\\\\.[^\\\\"]*)*)"', string_content)
        if len(samples) > 0:
            page_content = '\n'.join(samples)
            map_symboles = {'\\n': '\n', '\\t': '\t', '\\u2013': '-', '\\u2019': "'", '\\u00e9r': 'é', '\\u00f4me': 'ô'}
            for key, value in map_symboles.items():
                page_content = page_content.replace(key, value)
            disordered_pages.append({'page': path, 'page_content': page_content})
    pagexml_rels = [{'path': page_path, 'content': xmltodict.parse(zfile.read(f'visio/pages/_rels/{Path(page_path).stem}.xml.rels'))} for page_path in ordered_paths if f'visio/pages/_rels/{Path(page_path).stem}.xml.rels' in zfile.namelist()]
    ordered_pages: List[Tuple[int, str, str]] = []
    for page_number, (path, page_name) in enumerate(zip(ordered_paths, ordered_names)):
        relationships = self.get_relationships(path, zfile, ordered_paths, pagexml_rels)
        page_content = '\n'.join([page_['page_content'] for page_ in disordered_pages if page_['page'] in relationships] + [page_['page_content'] for page_ in disordered_pages if page_['page'] == path])
        ordered_pages.append((page_number, page_name, page_content))
    return ordered_pages