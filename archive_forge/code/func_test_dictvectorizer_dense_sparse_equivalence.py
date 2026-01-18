from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_dictvectorizer_dense_sparse_equivalence():
    """Check the equivalence between between sparse and dense DictVectorizer.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19978
    """
    movie_entry_fit = [{'category': ['thriller', 'drama'], 'year': 2003}, {'category': ['animation', 'family'], 'year': 2011}, {'year': 1974}]
    movie_entry_transform = [{'category': ['thriller'], 'unseen_feature': '3'}]
    dense_vectorizer = DictVectorizer(sparse=False)
    sparse_vectorizer = DictVectorizer(sparse=True)
    dense_vector_fit = dense_vectorizer.fit_transform(movie_entry_fit)
    sparse_vector_fit = sparse_vectorizer.fit_transform(movie_entry_fit)
    assert not sp.issparse(dense_vector_fit)
    assert sp.issparse(sparse_vector_fit)
    assert_allclose(dense_vector_fit, sparse_vector_fit.toarray())
    dense_vector_transform = dense_vectorizer.transform(movie_entry_transform)
    sparse_vector_transform = sparse_vectorizer.transform(movie_entry_transform)
    assert not sp.issparse(dense_vector_transform)
    assert sp.issparse(sparse_vector_transform)
    assert_allclose(dense_vector_transform, sparse_vector_transform.toarray())
    dense_inverse_transform = dense_vectorizer.inverse_transform(dense_vector_transform)
    sparse_inverse_transform = sparse_vectorizer.inverse_transform(sparse_vector_transform)
    expected_inverse = [{'category=thriller': 1.0}]
    assert dense_inverse_transform == expected_inverse
    assert sparse_inverse_transform == expected_inverse