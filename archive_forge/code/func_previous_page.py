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
def previous_page(self) -> None:
    if self.current_page > 0:
        self.current_page -= 1
        self.async_browse_stored_images()