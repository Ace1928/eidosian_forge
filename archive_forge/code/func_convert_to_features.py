from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset
def convert_to_features(self, example_batch):
    if self.print_text:
        print('Input Text: ', self.clean_text(example_batch['text']))
    input_ = example_batch['input']
    target_ = example_batch['target']
    prompt = f'Correct this to standard English: {input_}\n---\nCorrected: '
    prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
    label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)
    sample = {'input_ids': prompt_ids + label_ids, 'attention_mask': [1] * len(prompt_ids + label_ids), 'labels': [-100] * len(prompt_ids) + label_ids}
    return sample