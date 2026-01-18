import eventlet
from eventlet.hubs import get_hub
def on_read(d):
    original = ds[get_fileno(d)]['read']
    current.switch(([original], [], []))