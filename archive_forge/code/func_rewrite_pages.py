from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def rewrite_pages(self):
    pages_tree_nodes_to_delete = []
    for i, page_ref in enumerate(self.orig_pages):
        page_info = self.cached_objects[page_ref]
        del self.xref_table[page_ref.object_id]
        pages_tree_nodes_to_delete.append(page_info[PdfName(b'Parent')])
        if page_ref not in self.pages:
            continue
        stringified_page_info = {}
        for key, value in page_info.items():
            stringified_page_info[key.name_as_str()] = value
        stringified_page_info['Parent'] = self.pages_ref
        new_page_ref = self.write_page(None, **stringified_page_info)
        for j, cur_page_ref in enumerate(self.pages):
            if cur_page_ref == page_ref:
                self.pages[j] = new_page_ref
    for pages_tree_node_ref in pages_tree_nodes_to_delete:
        while pages_tree_node_ref:
            pages_tree_node = self.cached_objects[pages_tree_node_ref]
            if pages_tree_node_ref.object_id in self.xref_table:
                del self.xref_table[pages_tree_node_ref.object_id]
            pages_tree_node_ref = pages_tree_node.get(b'Parent', None)
    self.orig_pages = []