import argparse
import os
import orjson
from glob import glob
def convert_to_evalplus(results_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    for filename in glob(os.path.join(results_path, '*.json')):
        with open(filename, 'rb') as f:
            data = orjson.loads(f.read())
        result = bytearray()
        for item in data:
            if item['task_type'] == 'coding/humaneval':
                result.extend(orjson.dumps(item['answer']))
                result.extend(b'\n')
        with open(os.path.join(output_path, os.path.splitext(os.path.basename(filename))[0] + '.jsonl'), 'wb') as f:
            f.write(result)