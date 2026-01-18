def bubble(a):
    """
    Bubble Sort: compare adjacent elements of the list left-to-right,
    and swap them if they are out of order.  After one pass through
    the list swapping adjacent items, the largest item will be in
    the rightmost position.  The remainder is one element smaller;
    apply the same method to this list, and so on.
    """
    count = 0
    for i in range(len(a) - 1):
        for j in range(len(a) - i - 1):
            if a[j + 1] < a[j]:
                a[j], a[j + 1] = (a[j + 1], a[j])
                count += 1
    return count