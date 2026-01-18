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
def interactive_running(self, data):
    """
        Generate a model response.

        :param data:
            data to send to model

        :return:
            model act dictionary
        """
    reply = {}
    reply['text'] = data['personality'][0].decode()
    img_data = str(data['image'][0])
    _, encoded = img_data.split(',', 1)
    image = Image.open(io.BytesIO(b64decode(encoded))).convert('RGB')
    reply['image'] = SHARED['image_loader'].extract(image)
    SHARED['agent'].observe(reply)
    model_res = SHARED['agent'].act()
    return model_res