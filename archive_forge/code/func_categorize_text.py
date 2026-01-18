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
def categorize_text(self, text):
    categories = {'text': 0.0, 'code': 0.0, 'math': 0.0, 'logic': 0.0}
    text_classifier = self.TextCategorizer
    code_classifier = self.CodeCategorizer
    math_classifier = self.MathCategorizer
    logic_classifier = self.LogicCategorizer
    text_output = text_classifier.classify(text)
    code_output = code_classifier.classify(text)
    math_output = math_classifier.classify(text)
    logic_output = logic_classifier.classify(text)
    for category, weight in text_output.items():
        categories[category] += weight
    for category, weight in code_output.items():
        categories[category] += weight
    for category, weight in math_output.items():
        categories[category] += weight
    for category, weight in logic_output.items():
        categories[category] += weight
    total_weight = sum(categories.values())
    for category in categories:
        categories[category] /= total_weight
    return categories