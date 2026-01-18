import functools
def quoted_or_text(text, quoted_and_index):
    index = quoted_and_index[0]
    quoted_item = quoted_and_index[1]
    text += (', ' if len(selected) > 2 and (not index == len(selected) - 1) else ' ') + ('or ' if index == len(selected) - 1 else '') + quoted_item
    return text