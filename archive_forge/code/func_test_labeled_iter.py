from math import ceil
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
@pytest.mark.parametrize('max_iter', range(1, 5))
def test_labeled_iter(max_iter):
    st = SelfTrainingClassifier(KNeighborsClassifier(), max_iter=max_iter)
    st.fit(X_train, y_train_missing_labels)
    amount_iter_0 = len(st.labeled_iter_[st.labeled_iter_ == 0])
    assert amount_iter_0 == n_labeled_samples
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter