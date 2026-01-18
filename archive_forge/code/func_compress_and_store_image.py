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
def compress_and_store_image(self) -> None:
    """
        Initiates the process to compress and store the currently loaded image.
        """
    self.compress_btn.config(state='disabled')
    self.update_status('Compressing image...')
    self.progress_bar.start()
    future = asyncio.run_coroutine_threadsafe(self.async_compress_and_store_image(), self.asyncio_loop)
    future.add_done_callback(lambda x: self.after(0, self.on_compress_and_store_image_done, x))