import sqlite3
import os
from tqdm import tqdm
from collections import deque
import random
from parlai.core.teachers import create_task_agent_from_taskname
import parlai.utils.logging as logging
def store_contents(opt, task, save_path, context_length=-1, include_labels=True):
    """
    Preprocess and store a corpus of documents in sqlite.

    Args:
        task: ParlAI tasks of text (and possibly values) to store.
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)
    logging.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute('CREATE TABLE documents (id INTEGER PRIMARY KEY, text, value);')
    if not task:
        logging.info('No data to initialize table: just creating table.')
        logging.info('Add more data by passing observations to the agent.')
        logging.info('Committing...')
        conn.commit()
        conn.close()
        return
    ordered_opt = opt.copy()
    dt = opt.get('datatype', '').split(':')
    ordered_opt['datatype'] = ':'.join([dt[0], 'ordered'] + dt[1:])
    ordered_opt['batchsize'] = 1
    ordered_opt['task'] = task
    teacher = create_task_agent_from_taskname(ordered_opt)[0]
    episode_done = False
    current = []
    triples = []
    context_length = context_length if context_length >= 0 else None
    context = deque(maxlen=context_length)
    with tqdm(total=teacher.num_episodes()) as pbar:
        while not teacher.epoch_done():
            while not episode_done:
                action = teacher.act()
                current.append(action)
                episode_done = action['episode_done']
            for ex in current:
                if 'text' in ex:
                    text = ex['text']
                    context.append(text)
                    if len(context) > 1:
                        text = '\n'.join(context)
                labels = ex.get('labels', ex.get('eval_labels'))
                label = None
                if labels is not None:
                    label = random.choice(labels)
                    if include_labels:
                        context.append(label)
                triples.append((None, text, label))
            c.executemany('INSERT OR IGNORE INTO documents VALUES (?,?,?)', triples)
            pbar.update()
            episode_done = False
            triples.clear()
            current.clear()
            context.clear()
    logging.info('Read %d examples from %d episodes.' % (teacher.num_examples(), teacher.num_episodes()))
    logging.info('Committing...')
    conn.commit()
    conn.close()