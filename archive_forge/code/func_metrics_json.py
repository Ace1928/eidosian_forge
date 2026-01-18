import threading
import json
from gc import get_objects, garbage
from kivy.clock import Clock
from kivy.cache import Cache
from collections import OrderedDict
from kivy.logger import Logger
@app.route('/metrics.json')
def metrics_json():
    resp = make_response(json.dumps(metrics), 200)
    resp.headers['Content-Type'] = 'text/json'
    return resp