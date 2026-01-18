import time
from http.client import HTTPConnection
from subprocess import PIPE, Popen
import pytest
from playwright.sync_api import expect
@pytest.fixture()
def launch_jupyterlite():
    process = Popen(['python', '-m', 'http.server', '8123', '--directory', 'lite/dist/'], stdout=PIPE)
    retries = 5
    while retries > 0:
        conn = HTTPConnection('localhost:8123')
        try:
            conn.request('HEAD', 'index.html')
            response = conn.getresponse()
            if response is not None:
                break
        except ConnectionRefusedError:
            time.sleep(1)
            retries -= 1
    if not retries:
        raise RuntimeError('Failed to start http server')
    try:
        yield
    finally:
        process.terminate()
        process.wait()