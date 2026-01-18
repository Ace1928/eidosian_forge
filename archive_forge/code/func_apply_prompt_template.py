import copy
import datasets
def apply_prompt_template(sample):
    return {'prompt': prompt.format(dialog=sample['dialogue']), 'summary': sample['summary']}