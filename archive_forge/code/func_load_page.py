from typing import Any, Dict, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def load_page(self, page_summary: Dict[str, Any]) -> Document:
    """Read a page.

        Args:
            page_summary: Page summary from Notion API.
        """
    page_id = page_summary['id']
    metadata: Dict[str, Any] = {}
    for prop_name, prop_data in page_summary['properties'].items():
        prop_type = prop_data['type']
        if prop_type == 'rich_text':
            value = prop_data['rich_text'][0]['plain_text'] if prop_data['rich_text'] else None
        elif prop_type == 'title':
            value = prop_data['title'][0]['plain_text'] if prop_data['title'] else None
        elif prop_type == 'multi_select':
            value = [item['name'] for item in prop_data['multi_select']] if prop_data['multi_select'] else []
        elif prop_type == 'url':
            value = prop_data['url']
        elif prop_type == 'unique_id':
            value = f'{prop_data['unique_id']['prefix']}-{prop_data['unique_id']['number']}' if prop_data['unique_id'] else None
        elif prop_type == 'status':
            value = prop_data['status']['name'] if prop_data['status'] else None
        elif prop_type == 'people':
            value = [item['name'] for item in prop_data['people']] if prop_data['people'] else []
        elif prop_type == 'date':
            value = prop_data['date'] if prop_data['date'] else None
        elif prop_type == 'last_edited_time':
            value = prop_data['last_edited_time'] if prop_data['last_edited_time'] else None
        elif prop_type == 'created_time':
            value = prop_data['created_time'] if prop_data['created_time'] else None
        elif prop_type == 'checkbox':
            value = prop_data['checkbox']
        elif prop_type == 'email':
            value = prop_data['email']
        elif prop_type == 'number':
            value = prop_data['number']
        elif prop_type == 'select':
            value = prop_data['select']['name'] if prop_data['select'] else None
        else:
            value = None
        metadata[prop_name.lower()] = value
    metadata['id'] = page_id
    return Document(page_content=self._load_blocks(page_id), metadata=metadata)