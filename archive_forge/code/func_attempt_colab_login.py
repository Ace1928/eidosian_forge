import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def attempt_colab_login(app_url):
    """This renders an iframe to wandb in the hopes it posts back an api key."""
    from google.colab import output
    from google.colab._message import MessageError
    from IPython import display
    display.display(display.Javascript('\n        window._wandbApiKey = new Promise((resolve, reject) => {\n            function loadScript(url) {\n            return new Promise(function(resolve, reject) {\n                let newScript = document.createElement("script");\n                newScript.onerror = reject;\n                newScript.onload = resolve;\n                document.body.appendChild(newScript);\n                newScript.src = url;\n            });\n            }\n            loadScript("https://cdn.jsdelivr.net/npm/postmate/build/postmate.min.js").then(() => {\n            const iframe = document.createElement(\'iframe\')\n            iframe.style.cssText = "width:0;height:0;border:none"\n            document.body.appendChild(iframe)\n            const handshake = new Postmate({\n                container: iframe,\n                url: \'%s/authorize\'\n            });\n            const timeout = setTimeout(() => reject("Couldn\'t auto authenticate"), 5000)\n            handshake.then(function(child) {\n                child.on(\'authorize\', data => {\n                    clearTimeout(timeout)\n                    resolve(data)\n                });\n            });\n            })\n        });\n    ' % app_url.replace('http:', 'https:')))
    try:
        return output.eval_js('_wandbApiKey')
    except MessageError:
        return None