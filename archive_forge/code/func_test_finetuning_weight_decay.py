import pytest
from pytest import approx
from unittest.mock import patch
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from llama_recipes.finetuning import main
from llama_recipes.data.sampler import LengthBasedBatchSampler
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.AutoTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.get_peft_model')
@patch('llama_recipes.finetuning.StepLR')
def test_finetuning_weight_decay(step_lr, get_peft_model, get_dataset, tokenizer, get_model, train, mocker):
    kwargs = {'weight_decay': 0.01}
    get_dataset.return_value = get_fake_dataset()
    model = mocker.MagicMock(name='Model')
    model.parameters.return_value = [torch.ones(1, 1)]
    get_model.return_value = model
    main(**kwargs)
    assert train.call_count == 1
    args, kwargs = train.call_args
    optimizer = args[4]
    print(optimizer.state_dict())
    assert isinstance(optimizer, AdamW)
    assert optimizer.state_dict()['param_groups'][0]['weight_decay'] == approx(0.01)