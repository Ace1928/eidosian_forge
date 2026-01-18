import collections
import numpy as np
from numba.core import types
@wrap
def partition3(A, low, high):
    """
        Three-way partition [low, high) around a chosen pivot.
        A tuple (lt, gt) is returned such that:
            - all elements in [low, lt) are < pivot
            - all elements in [lt, gt] are == pivot
            - all elements in (gt, high] are > pivot
        """
    mid = low + high >> 1
    if LT(A[mid], A[low]):
        A[low], A[mid] = (A[mid], A[low])
    if LT(A[high], A[mid]):
        A[high], A[mid] = (A[mid], A[high])
    if LT(A[mid], A[low]):
        A[low], A[mid] = (A[mid], A[low])
    pivot = A[mid]
    A[low], A[mid] = (A[mid], A[low])
    lt = low
    gt = high
    i = low + 1
    while i <= gt:
        if LT(A[i], pivot):
            A[lt], A[i] = (A[i], A[lt])
            lt += 1
            i += 1
        elif LT(pivot, A[i]):
            A[gt], A[i] = (A[i], A[gt])
            gt -= 1
        else:
            i += 1
    return (lt, gt)