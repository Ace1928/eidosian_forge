from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
def save_local(self, folder_path: str, file_name: str='tfidf_vectorizer') -> None:
    try:
        import joblib
    except ImportError:
        raise ImportError('Could not import joblib, please install with `pip install joblib`.')
    path = Path(folder_path)
    path.mkdir(exist_ok=True, parents=True)
    joblib.dump(self.vectorizer, path / f'{file_name}.joblib')
    with open(path / f'{file_name}.pkl', 'wb') as f:
        pickle.dump((self.docs, self.tfidf_array), f)