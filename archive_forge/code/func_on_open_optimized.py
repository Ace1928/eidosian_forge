import websocket
import threading
import logging
def on_open_optimized(ws):

    def run(*args):
        ws.send('Optimized Hello from CodEVIE')
        threading.Timer(15, ws.close).start()
    threading.Thread(target=run).start()