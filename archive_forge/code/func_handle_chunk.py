import asyncio
import copy
import io
import os
import sys
import IPython
import traitlets
import ipyvuetify as v
def handle_chunk(self, content, buffer):
    content['buffer'] = buffer
    self.chunk_queue.append(content)