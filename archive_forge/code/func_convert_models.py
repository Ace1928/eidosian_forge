import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Tuple
import yaml
from tqdm import tqdm
from transformers.models.marian.convert_marian_to_pytorch import (
def convert_models(self, tatoeba_ids, dry_run=False):
    models_to_convert = [self.parse_metadata(x) for x in tatoeba_ids]
    save_dir = Path('marian_ckpt')
    dest_dir = Path(self.model_card_dir)
    dest_dir.mkdir(exist_ok=True)
    for model in tqdm(models_to_convert):
        if 'SentencePiece' not in model['pre-processing']:
            print(f"Skipping {model['release']} because it doesn't appear to use SentencePiece")
            continue
        if not os.path.exists(save_dir / model['_name']):
            download_and_unzip(f'{TATOEBA_MODELS_URL}/{model['release']}', save_dir / model['_name'])
        opus_language_groups_to_hf = convert_opus_name_to_hf_name
        pair_name = opus_language_groups_to_hf(model['_name'])
        convert(save_dir / model['_name'], dest_dir / f'opus-mt-{pair_name}')
        self.write_model_card(model, dry_run=dry_run)