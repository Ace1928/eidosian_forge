import eventlet
from eventlet.hubs import get_hub
def on_write(d):
    original = ds[get_fileno(d)]['write']
    current.switch(([], [original], []))