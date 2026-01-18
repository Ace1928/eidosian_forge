import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
def load_datasets(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Tuple[UnmixDataset, UnmixDataset, argparse.Namespace]:
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """
    if args.dataset == 'aligned':
        parser.add_argument('--input-file', type=str)
        parser.add_argument('--output-file', type=str)
        args = parser.parse_args()
        args.target = Path(args.output_file).stem
        dataset_kwargs = {'root': Path(args.root), 'seq_duration': args.seq_dur, 'input_file': args.input_file, 'output_file': args.output_file}
        args.target = Path(args.output_file).stem
        train_dataset = AlignedDataset(split='train', random_chunks=True, **dataset_kwargs)
        valid_dataset = AlignedDataset(split='valid', **dataset_kwargs)
    elif args.dataset == 'sourcefolder':
        parser.add_argument('--interferer-dirs', type=str, nargs='+')
        parser.add_argument('--target-dir', type=str)
        parser.add_argument('--ext', type=str, default='.wav')
        parser.add_argument('--nb-train-samples', type=int, default=1000)
        parser.add_argument('--nb-valid-samples', type=int, default=100)
        parser.add_argument('--source-augmentations', type=str, nargs='+')
        args = parser.parse_args()
        args.target = args.target_dir
        dataset_kwargs = {'root': Path(args.root), 'interferer_dirs': args.interferer_dirs, 'target_dir': args.target_dir, 'ext': args.ext}
        source_augmentations = aug_from_str(args.source_augmentations)
        train_dataset = SourceFolderDataset(split='train', source_augmentations=source_augmentations, random_chunks=True, nb_samples=args.nb_train_samples, seq_duration=args.seq_dur, **dataset_kwargs)
        valid_dataset = SourceFolderDataset(split='valid', random_chunks=True, seq_duration=args.seq_dur, nb_samples=args.nb_valid_samples, **dataset_kwargs)
    elif args.dataset == 'trackfolder_fix':
        parser.add_argument('--target-file', type=str)
        parser.add_argument('--interferer-files', type=str, nargs='+')
        parser.add_argument('--random-track-mix', action='store_true', default=False, help='Apply random track mixing augmentation')
        parser.add_argument('--source-augmentations', type=str, nargs='+')
        args = parser.parse_args()
        args.target = Path(args.target_file).stem
        dataset_kwargs = {'root': Path(args.root), 'interferer_files': args.interferer_files, 'target_file': args.target_file}
        source_augmentations = aug_from_str(args.source_augmentations)
        train_dataset = FixedSourcesTrackFolderDataset(split='train', source_augmentations=source_augmentations, random_track_mix=args.random_track_mix, random_chunks=True, seq_duration=args.seq_dur, **dataset_kwargs)
        valid_dataset = FixedSourcesTrackFolderDataset(split='valid', seq_duration=None, **dataset_kwargs)
    elif args.dataset == 'trackfolder_var':
        parser.add_argument('--ext', type=str, default='.wav')
        parser.add_argument('--target-file', type=str)
        parser.add_argument('--source-augmentations', type=str, nargs='+')
        parser.add_argument('--random-interferer-mix', action='store_true', default=False, help='Apply random interferer mixing augmentation')
        parser.add_argument('--silence-missing', action='store_true', default=False, help='silence missing targets')
        args = parser.parse_args()
        args.target = Path(args.target_file).stem
        dataset_kwargs = {'root': Path(args.root), 'target_file': args.target_file, 'ext': args.ext, 'silence_missing_targets': args.silence_missing}
        source_augmentations = Compose([globals()['_augment_' + aug] for aug in args.source_augmentations])
        train_dataset = VariableSourcesTrackFolderDataset(split='train', source_augmentations=source_augmentations, random_interferer_mix=args.random_interferer_mix, random_chunks=True, seq_duration=args.seq_dur, **dataset_kwargs)
        valid_dataset = VariableSourcesTrackFolderDataset(split='valid', seq_duration=None, **dataset_kwargs)
    else:
        parser.add_argument('--is-wav', action='store_true', default=False, help='loads wav instead of STEMS')
        parser.add_argument('--samples-per-track', type=int, default=64)
        parser.add_argument('--source-augmentations', type=str, default=['gain', 'channelswap'], nargs='+')
        args = parser.parse_args()
        dataset_kwargs = {'root': args.root, 'is_wav': args.is_wav, 'subsets': 'train', 'target': args.target, 'download': args.root is None, 'seed': args.seed}
        source_augmentations = aug_from_str(args.source_augmentations)
        train_dataset = MUSDBDataset(split='train', samples_per_track=args.samples_per_track, seq_duration=args.seq_dur, source_augmentations=source_augmentations, random_track_mix=True, **dataset_kwargs)
        valid_dataset = MUSDBDataset(split='valid', samples_per_track=1, seq_duration=None, **dataset_kwargs)
    return (train_dataset, valid_dataset, args)