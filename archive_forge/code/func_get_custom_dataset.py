import copy
import datasets
import itertools
def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('OpenAssistant/oasst1', split=split)
    dataset = dataset.map(lambda sample: {'message_id': sample['message_id'], 'parent_id': sample['parent_id'], 'text': sample['text']}, batched=True, remove_columns=list(dataset.features))
    nodes = {}
    messages = {}
    root_ids = []
    for data in dataset:
        if data['parent_id']:
            nodes[data['parent_id']] = nodes.get(data['parent_id'], []) + [data['message_id']]
        else:
            root_ids.append(data['message_id'])
        messages[data['message_id']] = data['text']

    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]

    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads
    dataset = dataset.filter(lambda x: x['message_id'] in root_ids)
    dataset = dataset.map(lambda x: {'thread': get_threads_from_root(x['message_id'])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {'thread': [i for row in x['thread'] for i in row]}, batched=True)

    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': content})
        return {'dialog': dialog}
    dataset = dataset.map(lambda x: to_dialog(x['thread']), remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenize_dialog(x['dialog'], tokenizer), remove_columns=list(dataset.features))
    return dataset