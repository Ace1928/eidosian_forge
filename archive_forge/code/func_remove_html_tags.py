from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def remove_html_tags(self, html_string: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup.get_text()