import asyncio
import cProfile
import hashlib
import io
import itertools
import json
import logging
import resource
import os
import pstats
import queue
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from functools import reduce
from logging.handlers import MemoryHandler, RotatingFileHandler
from logging import StreamHandler, FileHandler
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator, Union, Callable
import coloredlogs
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import (
from wordcloud import WordCloud
from tqdm import tqdm
import requests
from transformers import BertModel, BertTokenizer
import functools
def setup_menu(self):
    """
        Sets up the menu for the GUI, providing options for file operations, processing, visualization, and exit, enhanced with modern aesthetics.
        """
    menu_bar = tk.Menu(self.root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label='Open', command=lambda: self.thread_action(self.file_browse))
    file_menu.add_command(label='Save', command=lambda: self.thread_action(self.file_save))
    file_menu.add_command(label='Open Multiple', command=lambda: self.thread_action(self.file_browse_multiple))
    file_menu.add_command(label='Open Directory', command=lambda: self.thread_action(self.directory_browse))
    file_menu.add_command(label='Save Session', command=self.data_manager.save_session_data)
    file_menu.add_command(label='Load Session', command=self.data_manager.load_session_data)
    file_menu.add_separator()
    file_menu.add_command(label='Exit', command=self.root.quit)
    menu_bar.add_cascade(label='File', menu=file_menu)
    self.root.config(menu=menu_bar)
    advanced_logger.log(logging.DEBUG, 'Menu setup completed with file operations, processing, visualization, and exit functionality.')