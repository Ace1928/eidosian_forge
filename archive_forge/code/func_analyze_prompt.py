import csv
import json
import time
import random
import threading
import numpy as np
import requests
import transformers
import torch
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
def analyze_prompt(input):
    start_time = time.time()
    key = ''
    endpoint = ''
    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
    request = AnalyzeTextOptions(text=input)
    try:
        response = client.analyze_text(request)
    except HttpResponseError as e:
        print('prompt failed due to content safety filtering.')
        if e.error:
            print(f'Error code: {e.error.code}')
            print(f'Error message: {e.error.message}')
            raise
        print(e)
        raise
    analyze_end_time = time.time()
    analyze_latency = (analyze_end_time - start_time) * 1000