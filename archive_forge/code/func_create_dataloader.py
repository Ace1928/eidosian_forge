import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import os
import json
from collections import defaultdict, deque
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math
import tensorboard
from torch.utils.tensorboard import SummaryWriter
def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """Create DataLoader for better batch processing and data handling."""
    try:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader
    except Exception as e:
        print(f'Failed to create DataLoader: {str(e)}')
        return None