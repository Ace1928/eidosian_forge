import websocket
import threading
import logging
def on_message_optimized(ws, message):
    logging.info(f'WebSocket message: {message}')