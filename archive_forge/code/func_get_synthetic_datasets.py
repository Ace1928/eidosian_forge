from collections import namedtuple
from distutils.version import LooseVersion
import io
import operator
import tempfile
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
def get_synthetic_datasets():
    lm_dataset = torch.randint(1, 10000, (2049990,))
    return DatasetsInfo(10000, lm_dataset, lm_dataset, lm_dataset)