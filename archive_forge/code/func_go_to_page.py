import time
from sys import platform
from typing import (
def go_to_page(self, url: str) -> None:
    self.page.goto(url=url if '://' in url else 'http://' + url)
    self.client = self.page.context.new_cdp_session(self.page)
    self.page_element_buffer = {}