import collections
import os
from xml.etree.ElementTree import Element as ET_Element
from .vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
@staticmethod
def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(VOCDetection.parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict