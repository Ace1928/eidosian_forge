import websocket
import threading
import logging
def on_error_optimized(ws, error):
    logging.error(f'WebSocket error: {error}')