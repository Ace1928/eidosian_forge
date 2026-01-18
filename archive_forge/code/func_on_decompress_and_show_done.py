import asyncio
import io
import tkinter as tk
import threading
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from cryptography.fernet import Fernet
from core_services import LoggingManager, EncryptionManager
from database import (
from image_processing import (
def on_decompress_and_show_done(self, future):
    """
        Callback function to be called when async_decompress_and_show coroutine is done.
        """
    try:
        future.result()
        self.update_status('Image decompressed successfully.')
    except Exception as e:
        LoggingManager.error(f'Failed to decompress image: {e}')
        self.update_status('Failed to decompress image. Please try again.', error=True)
    finally:
        self.progress_bar.stop()