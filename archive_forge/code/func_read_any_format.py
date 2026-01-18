from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def read_any_format(fname: str, fields: list[str]=['prompt', 'completion']) -> tuple[pd.DataFrame | None, Remediation]:
    """
    This function will read a file saved in .csv, .json, .txt, .xlsx or .tsv format using pandas.
     - for .xlsx it will read the first sheet
     - for .txt it will assume completions and split on newline
    """
    remediation = None
    necessary_msg = None
    immediate_msg = None
    error_msg = None
    df = None
    if os.path.isfile(fname):
        try:
            if fname.lower().endswith('.csv') or fname.lower().endswith('.tsv'):
                file_extension_str, separator = ('CSV', ',') if fname.lower().endswith('.csv') else ('TSV', '\t')
                immediate_msg = f'\n- Based on your file extension, your file is formatted as a {file_extension_str} file'
                necessary_msg = f'Your format `{file_extension_str}` will be converted to `JSONL`'
                df = pd.read_csv(fname, sep=separator, dtype=str).fillna('')
            elif fname.lower().endswith('.xlsx'):
                immediate_msg = '\n- Based on your file extension, your file is formatted as an Excel file'
                necessary_msg = 'Your format `XLSX` will be converted to `JSONL`'
                xls = pd.ExcelFile(fname)
                sheets = xls.sheet_names
                if len(sheets) > 1:
                    immediate_msg += '\n- Your Excel file contains more than one sheet. Please either save as csv or ensure all data is present in the first sheet. WARNING: Reading only the first sheet...'
                df = pd.read_excel(fname, dtype=str).fillna('')
            elif fname.lower().endswith('.txt'):
                immediate_msg = '\n- Based on your file extension, you provided a text file'
                necessary_msg = 'Your format `TXT` will be converted to `JSONL`'
                with open(fname, 'r') as f:
                    content = f.read()
                    df = pd.DataFrame([['', line] for line in content.split('\n')], columns=fields, dtype=str).fillna('')
            elif fname.lower().endswith('.jsonl'):
                df = pd.read_json(fname, lines=True, dtype=str).fillna('')
                if len(df) == 1:
                    immediate_msg = '\n- Your JSONL file appears to be in a JSON format. Your file will be converted to JSONL format'
                    necessary_msg = 'Your format `JSON` will be converted to `JSONL`'
                    df = pd.read_json(fname, dtype=str).fillna('')
                else:
                    pass
            elif fname.lower().endswith('.json'):
                try:
                    df = pd.read_json(fname, lines=True, dtype=str).fillna('')
                    if len(df) == 1:
                        df = pd.read_json(fname, dtype=str).fillna('')
                    else:
                        immediate_msg = '\n- Your JSON file appears to be in a JSONL format. Your file will be converted to JSONL format'
                        necessary_msg = 'Your format `JSON` will be converted to `JSONL`'
                except ValueError:
                    df = pd.read_json(fname, dtype=str).fillna('')
            else:
                error_msg = 'Your file must have one of the following extensions: .CSV, .TSV, .XLSX, .TXT, .JSON or .JSONL'
                if '.' in fname:
                    error_msg += f' Your file `{fname}` ends with the extension `.{fname.split('.')[-1]}` which is not supported.'
                else:
                    error_msg += f' Your file `{fname}` is missing a file extension.'
        except (ValueError, TypeError):
            file_extension_str = fname.split('.')[-1].upper()
            error_msg = f'Your file `{fname}` does not appear to be in valid {file_extension_str} format. Please ensure your file is formatted as a valid {file_extension_str} file.'
    else:
        error_msg = f'File {fname} does not exist.'
    remediation = Remediation(name='read_any_format', necessary_msg=necessary_msg, immediate_msg=immediate_msg, error_msg=error_msg)
    return (df, remediation)