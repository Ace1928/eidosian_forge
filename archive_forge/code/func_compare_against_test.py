import os
from typing import List
def compare_against_test(base_filename: str, feature_filename: str, parser_only: bool, secondary_filename: str=None):
    """
    Tests whether the additional code inside of `feature_filename` was implemented in `base_filename`. This should be
    used when testing to see if `complete_*_.py` examples have all of the implementations from each of the
    `examples/by_feature/*` scripts.

    It utilizes `nlp_example.py` to extract out all of the repeated training code, so that only the new additional code
    is examined and checked. If something *other* than `nlp_example.py` should be used, such as `cv_example.py` for the
    `complete_cv_example.py` script, it should be passed in for the `secondary_filename` parameter.

    Args:
        base_filename (`str` or `os.PathLike`):
            The filepath of a single "complete" example script to test, such as `examples/complete_cv_example.py`
        feature_filename (`str` or `os.PathLike`):
            The filepath of a single feature example script. The contents of this script are checked to see if they
            exist in `base_filename`
        parser_only (`bool`):
            Whether to compare only the `main()` sections in both files, or to compare the contents of
            `training_loop()`
        secondary_filename (`str`, *optional*):
            A potential secondary filepath that should be included in the check. This function extracts the base
            functionalities off of "examples/nlp_example.py", so if `base_filename` is a script other than
            `complete_nlp_example.py`, the template script should be included here. Such as `examples/cv_example.py`
    """
    with open(base_filename) as f:
        base_file_contents = f.readlines()
    with open(os.path.abspath(os.path.join('examples', 'nlp_example.py'))) as f:
        full_file_contents = f.readlines()
    with open(feature_filename) as f:
        feature_file_contents = f.readlines()
    if secondary_filename is not None:
        with open(secondary_filename) as f:
            secondary_file_contents = f.readlines()
    if parser_only:
        base_file_func = clean_lines(get_function_contents_by_name(base_file_contents, 'main'))
        full_file_func = clean_lines(get_function_contents_by_name(full_file_contents, 'main'))
        feature_file_func = clean_lines(get_function_contents_by_name(feature_file_contents, 'main'))
        if secondary_filename is not None:
            secondary_file_func = clean_lines(get_function_contents_by_name(secondary_file_contents, 'main'))
    else:
        base_file_func = clean_lines(get_function_contents_by_name(base_file_contents, 'training_function'))
        full_file_func = clean_lines(get_function_contents_by_name(full_file_contents, 'training_function'))
        feature_file_func = clean_lines(get_function_contents_by_name(feature_file_contents, 'training_function'))
        if secondary_filename is not None:
            secondary_file_func = clean_lines(get_function_contents_by_name(secondary_file_contents, 'training_function'))
    _dl_line = 'train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)\n'
    new_feature_code = []
    passed_idxs = []
    it = iter(feature_file_func)
    for i in range(len(feature_file_func) - 1):
        if i not in passed_idxs:
            line = next(it)
            if line not in full_file_func and line.lstrip() != _dl_line:
                if 'TESTING_MOCKED_DATALOADERS' not in line:
                    new_feature_code.append(line)
                    passed_idxs.append(i)
                else:
                    _ = next(it)
    new_full_example_parts = []
    passed_idxs = []
    for i, line in enumerate(base_file_func):
        if i not in passed_idxs:
            if line not in full_file_func and line.lstrip() != _dl_line:
                if 'TESTING_MOCKED_DATALOADERS' not in line:
                    new_full_example_parts.append(line)
                    passed_idxs.append(i)
    diff_from_example = [line for line in new_feature_code if line not in new_full_example_parts]
    if secondary_filename is not None:
        diff_from_two = [line for line in full_file_contents if line not in secondary_file_func]
        diff_from_example = [line for line in diff_from_example if line not in diff_from_two]
    return diff_from_example