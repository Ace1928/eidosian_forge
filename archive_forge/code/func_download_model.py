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
def download_model(self) -> None:
    """
        Download the model and tokenizer only if they are not already downloaded.
        """
    model_path: Path = Path(f'{self.model_path}/{self.model_name}')
    tokenizer_path: Path = Path(f'{self.model_path}/{self.model_name}-tokenizer')
    if not model_path.exists():
        advanced_logger.log(logging.INFO, f'Downloading model from {self.model_name}...')
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.save_pretrained(model_path)
        advanced_logger.log(logging.INFO, f'Model saved at {model_path}')
    else:
        advanced_logger.log(logging.INFO, f'Model already exists at {model_path}. Loading model...')
        self.model = BertModel.from_pretrained(model_path)
    if not tokenizer_path.exists():
        advanced_logger.log(logging.INFO, f'Downloading tokenizer from {self.model_name}...')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.tokenizer.save_pretrained(tokenizer_path)
        advanced_logger.log(logging.INFO, f'Tokenizer saved at {tokenizer_path}')
    else:
        advanced_logger.log(logging.INFO, f'Tokenizer already exists at {tokenizer_path}. Loading tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    advanced_logger.log(logging.INFO, f'Model and tokenizer are ready at {self.model_path}')