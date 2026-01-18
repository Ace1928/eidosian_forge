from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.image_featurizers import ImageLoader
from typing import Dict, Any
import json
import cgi
import PIL.Image as Image
from base64 import b64decode
import io
import os
def handle_http(self, status_code, path, text=None):
    """
        Generate HTTP.
        """
    self.send_response(status_code)
    self.send_header('Content-type', 'text/html')
    self.end_headers()
    content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
    return bytes(content, 'UTF-8')